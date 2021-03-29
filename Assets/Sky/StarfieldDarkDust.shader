
Shader "Skybox/StarfieldDarkDust"
{
    Properties
    {
        _MainTex ("tex2D", 2D) = "white" {}
    }
    SubShader
    {
        Tags { "RenderType"="Opaque" }
        LOD 100

        Pass
        {
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            // make fog work
            #pragma multi_compile_fog

            #include "UnityCG.cginc"

// variant of https://shadertoy.com/view/wscfDr

#define rot(a)    float2x2( cos(a+float4(0,11,33,0)) )                              // rotation  

            uniform sampler2D _MainTex; 

            struct appdata
            {
                float4 vertex : POSITION;
                float2 uv : TEXCOORD0;
            };

            struct v2f {
                float4 uv : TEXCOORD0;         //posWorld
                float4 vertex : SV_POSITION;   //pos
            };

            v2f vert (appdata v) {
                appdata v2;
                v2.vertex = v.vertex; //mul(v.vertex ,float4x4(-1,0.,0.,0.,  0.,1.,0.,0.,  0.,0.,1.,0.,  0.,0.,0.,1.));
                v2f o = (v2f)0;
                o.uv = mul(unity_ObjectToWorld, v2.vertex);
                o.vertex = UnityObjectToClipPos( v.vertex); // * float4(1.0,1.,1.0,1.) ); // squish skybox sphere
                return o;
            }



         fixed4 frag (v2f v) : SV_Target
            {
                float2 U = v.vertex;

                float3 viewDirection = normalize(v.uv.xyz- _WorldSpaceCameraPos.xyz  );
                fixed4 O = tex2D(_MainTex, v.uv);
                
                float3 rd = viewDirection;                                                        // ray direction for fragCoord.xy
                float3 ro = _WorldSpaceCameraPos.xyz*.0001;                                             // ray origin

    float d,l, I;
    float3  R = float3(800,450,0), //iResolution, 
          M = float3(8,4,0)/1e2*cos(_Time.y+float3(0,11,0)),                //iMouse.z > 0. ? iMouse.xyz/R -.5: 
          D = rd, //normalize(float3( U+U, -3.5*R.y ) - R ),                          // ray direction
          p = 2./R, L; 
    float4  r, T = float4(1,1,1,1), 
          C = exp(-.5*float4(1,2,4,0)),                                         // dark dust unit transparency
          S = float4(.4,.6,1,0);                                                // star color
    O-=O;
    p.xy = mul(p.xy,rot( .5+6.*M.y)), D.xy = mul(D.xy,rot( .5+6.*M.y));                         // camera move 
    p.xz = mul(p.xz,rot( 2.-6.*M.x)), D.xz = mul(D.xz,rot( 2.-6.*M.x));
    p.xy += cos( _Time.y + float2(0,11)) - 2.*_Time.y;
    
    for ( float i=1.; i < 150.; i++,  p += D ) {                              // volume ray-casting
        r = tex2D( _MainTex, (floor(p)+.5)/32. );                          // per voxel 4 random floats
        // NB: we should either visit neighbor cells or avoid cell margins, but cut spheres are rare
        L = frac(p) - r.rgb ;                                                // distance to sphere in voxel
        d = -dot(L,D);            // distance ray-sphere: d/dl ( || P+l.D - C ||² )  = 2l + 2 (CP.D)
        if (// r.a > .9 &&        // less blue-noise more white noise. but strange artifact
            i>0. || d > 0.)                                                   // for not seeing in our back ;-)
#define PSF(L) ( l = i/length(L) ,  min( 1e9, l*l*l ) ) // light on sensor = Istar * 1/d² * 1/pixeldist³ = Istar * d/dist³, cf https://www.shadertoy.com/results?query=star+psf
            I = PSF( L += d*D ),                                              // L= point on the ray that is closest to sphere
#if 0                             // 1 to see star cross
            L.xz *= rot( -(2.-6.*M.x) ), L.xy *= rot( -(.5+6.*M.y) );         // back to screen-space
            I +=  .3* PSF( L.xy * float2(10,.2) ),
            I +=  .3* PSF( L.xy * float2(.2,10) ),
#endif
            O += T *   S * I/(i*i);                                           // blend to final color                     

        l = max(0., 3.*tex2D( _MainTex, p/128. ).a -2.);                   // random density dark dust transparency
        T *= pow(C, float4(l,l,l,l));                                                 // cumul opacity
    }
    O = pow(  9e-7* O, float4(1./2.2,1./2.2,1./2.2,1./2.2) );                                        // exposure + to sRGB


                return O;
            }

            ENDCG
        }
    }
}

                
