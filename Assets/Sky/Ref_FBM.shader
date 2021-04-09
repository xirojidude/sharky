
Shader "Skybox/Empty"
{
    Properties
    {
        _MainTex ("Texture", 2D) = "white" {}
        _SunDir ("Sun Dir", Vector) = (-.11,.07,0.99,0) 
        _XYZPos ("XYZ Offset", Vector) = (0, 15, -.25 ,0) 
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

            uniform sampler2D _MainTex; 
            float4 _SunDir,_XYZPos;

            struct appdata
            {
                float4 vertex : POSITION;
                float2 uv : TEXCOORD0;
            };

            struct v2f {
                float4 uv : TEXCOORD0;         //posWorld
                float4 worldPos : TEXCOORD1;
                float4 vertex : SV_POSITION;   //pos
                float4 screenPos : TEXCOORD2;
            };

            v2f vert (appdata v) {
                appdata v2;
                v2.vertex = v.vertex; //mul(v.vertex ,float4x4(-1,0.,0.,0.,  0.,1.,0.,0.,  0.,0.,1.,0.,  0.,0.,0.,1.));
                v2f o = (v2f)0;
                o.uv = mul(unity_ObjectToWorld, v2.vertex);
                o.vertex = UnityObjectToClipPos( v.vertex); // * float4(1.0,1.,1.0,1.) ); // squish skybox sphere
                o.worldPos = mul(unity_ObjectToWorld, v2.vertex);
                o.screenPos  = ComputeScreenPos(o.vertex);
                //o.screenPos.z = -(mul(UNITY_MATRIX_MV, vertex).z * _ProjectionParams.w);     // old calculation, as I used the depth buffer comparision for min max ray march. 

                return o;
            }



         fixed4 frag (v2f v) : SV_Target
            {
                float2 fragCoord = v.vertex;
                float2 screenUV = v.screenPos.xy / v.screenPos.w;

                float3 viewDirection = normalize(v.uv.xyz- _WorldSpaceCameraPos.xyz  );
                fixed4 fragColor = tex2D(_MainTex, v.uv);
                
                float3 rd = viewDirection;                                                        // ray direction for fragCoord.xy
                float3 ro = _WorldSpaceCameraPos.xyz+ _XYZPos;                                             // ray origin



                return fragColor;
            }

            ENDCG
        }
    }
}

//------------------------------------------------------------------------
// Alps
float FBM( float3 p )
{
    
    p *= .015;
    p.xz *= .3;
    //p.zy -= _Time.y * .04;
    
    float f;
    f  = 0.5000  * Noise(p); p = p * 3.02; //p.y -= gTime*.2;
    f += 0.2500  * Noise(p); p = p * 3.03; //p.y += gTime*.06;
    f += 0.1250  * Noise(p); p = p * 4.01;
    f += 0.0625  * Noise(p); p = p * 4.023;
    //f += 0.03125 * Noise(p);
    return f;
}

//------------------------------------------------------------------------
// DesertCanyon
// Simple fBm to produce some clouds.
float fbm(in float3 p){
    
    // Four layers of 3D noise.
    return .5333*noise3D(p) + .2667*noise3D(p*2.02) + .1333*noise3D(p*4.03) + .0667*noise3D(p*8.03);

}

//------------------------------------------------------------------------
// DesertSand
// Gradient noise fBm.
float fBm(in float2 p){
    
    return gradN2D(p)*.57 + gradN2D(p*2.)*.28 + gradN2D(p*4.)*.15;
    
}


//------------------------------------------------------------------------
// GalaxyNavigator
// fractional Brownian motion
float fbmslow( float3 p )
{
  float f = 0.5000*noise( p ); p = mul(m,p*1.2);
  f += 0.2500*noise( p ); p = mul(m,p*1.3);
  f += 0.1666*noise( p ); p = mul(m,p*1.4);
  f += 0.0834*noise( p ); p = mul(m,p*1.84);
  return f;
}

float fbm( float3 p )
{
  float f = 0., a = 1., s=0.;
  f += a*noise( p ); p = mul(m,p*1.149); s += a; a *= .75;
  f += a*noise( p ); p = mul(m,p*1.41); s += a; a *= .75;
  f += a*noise( p ); p = mul(m,p*1.51); s += a; a *= .65;
  f += a*noise( p ); p = mul(m,p*1.21); s += a; a *= .35;
  f += a*noise( p ); p = mul(m,p*1.41); s += a; a *= .75;
  f += a*noise( p ); 
  return f/s;
}



//------------------------------------------------------------------------
// JetStream
float fbm(float3 p)
{
    float3 q = p;
    //q.xy = rotate(p.xy, _Time.y);
    
    p += (nse3d(p * 3.0) - 0.5) * 0.3;
    
    //float v = nse3d(p) * 0.5 + nse3d(p * 2.0) * 0.25 + nse3d(p * 4.0) * 0.125 + nse3d(p * 8.0) * 0.0625;
    
    //p.y += _Time.y * 0.2;
    
    float mtn = _Time.y * 0.15;
    
    float v = 0.0;
    float fq = 1.0, am = 0.5;
    for(int i = 0; i < 6; i++)
    {
        v += nse3d(p * fq + mtn * fq) * am;
        fq *= 2.0;
        am *= 0.5;
    }
    return v;
}

float fbmHQ(float3 p)
{
    float3 q = p;
    q.xy = rotate(p.xy, _Time.y);
    
    p += (nse3d(p * 3.0) - 0.5) * 0.4;
    
    //float v = nse3d(p) * 0.5 + nse3d(p * 2.0) * 0.25 + nse3d(p * 4.0) * 0.125 + nse3d(p * 8.0) * 0.0625;
    
    //p.y += _Time.y * 0.2;
    
    float mtn = _Time.y * 0.2;
    
    float v = 0.0;
    float fq = 1.0, am = 0.5;
    for(int i = 0; i < 9; i++)
    {
        v += nse3d(p * fq + mtn * fq) * am;
        fq *= 2.0;
        am *= 0.5;
    }
    return v;
}
