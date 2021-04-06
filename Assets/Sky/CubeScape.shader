
Shader "Skybox/CubeScape"
{
    Properties
    {
        _MainTex ("tex2D", 2D) = "white" {}
        _SunDir ("Sun Dir", Vector) = (-.11,.07,0.99,0) 
        _XYZPos ("XYZ Offset", Vector) = (0, 15, -.25 ,0) 
        _Beat ("_Beat", float) = 0.0
        _Volume ("_Volume", float) = 0.0

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
            uniform int _SoundArrayLength = 256;
            float _Beat;
            uniform float _SoundArray[256];
            uniform float _Volume;

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

// Created by inigo quilez - iq/2013
// I share this piece (art and code) here in Shadertoy and through its Public API, only for educational purposes. 
// You cannot use, sell, share or host this piece or modifications of it as part of your own commercial or non-commercial product, website or project.
// You can share a link to it or an unmodified screenshot of it provided you attribute "by Inigo Quilez, @iquilezles and iquilezles.org". 
// If you are a teacher, lecturer, educator or similar and these conditions are too restrictive for your needs, please contact me and we'll work it out.

//---------------------------------

#if HW_PERFORMANCE==0
#define AA 1
#else
#define AA 2   // make this 2 or 3 for antialiasing
#endif

float hash( float n ) { return frac(sin(n)*13.5453123); }

float maxcomp( in float3 v ) { return max( max( v.x, v.y ), v.z ); }

float dbox( float3 p, float3 b, float r )
{
    return length(max(abs(p)-b,0.0))-r;
}

float4 texcube( sampler2D sam, in float3 p, in float3 n )
{
    float3 a = 1;//n*n;
    float4 x = tex2D( sam, p.yz );
    float4 y = tex2D( sam, p.zx );
    float4 z = tex2D( sam, p.yx );
    return (x*a.x + y*a.y + z*a.z) / (a.x + a.y + a.z);
}

//---------------------------------

float freqs[4];

float3 mapH( in float2 pos )
{
    float2 fpos = frac( pos ); 
    float2 ipos = floor( pos );
    
    float f = 0.0;  
    float id = hash( ipos.x + ipos.y*57.0 );
    f += freqs[0] * clamp(1.0 - abs(id-0.20)/0.30, 0.0, 1.0 );
    f += freqs[1] * clamp(1.0 - abs(id-0.40)/0.30, 0.0, 1.0 );
    f += freqs[2] * clamp(1.0 - abs(id-0.60)/0.30, 0.0, 1.0 );
    f += freqs[3] * clamp(1.0 - abs(id-0.80)/0.30, 0.0, 1.0 );

    f = pow( clamp( f, 0.0, 1.0 ), 2.0 );
    float h = 2.5*f;

    return float3( h, id, f );
}

float3 map( in float3 pos )
{
    float2  p = frac( pos.xz ); 
    float3  m = mapH( pos.xz );
    float d = dbox( float3(p.x-0.5,pos.y-0.5*m.x,p.y-0.5), float3(0.3,m.x*0.5,0.3), 0.1 );
    return float3( d, m.yz );
}

const float surface = 0.001;

float3 trace( float3 ro, in float3 rd, in float tmin, in float tmax )
{
    ro += tmin*rd;
    
    float2 pos = floor(ro.xz);
    float3 rdi = 1.0/rd;
    float3 rda = abs(rdi);
    float2 rds = sign(rd.xz);
    float2 dis = (pos-ro.xz+ 0.5 + rds*0.5) * rdi.xz;
    
    float3 res = float3( -1.0,-1,-1  );

    // traverse regular grid (in 2D)
    float2 mm = float2(0.0,0);
    for( int i=0; i<28; i++ ) 
    {
        float3 cub = mapH( pos );

        #if 1
            float2 pr = pos+0.5-ro.xz;
            float2 mini = (pr-0.5*rds)*rdi.xz;
            float s = max( mini.x, mini.y );
            if( (tmin+s)>tmax ) break;
        #endif
        
        
        // intersect box
        float3  ce = float3( pos.x+0.5, 0.5*cub.x, pos.y+0.5 );
        float3  rb = float3(0.3,cub.x*0.5,0.3);
        float3  ra = rb + 0.12;
        float3  rc = ro - ce;
        float tN = maxcomp( -rdi*rc - rda*ra );
        float tF = maxcomp( -rdi*rc + rda*ra );
        if( tN < tF )//&& tF > 0.0 )
        {
            // raymarch
            float s = tN;
            float h = 1.0;
            for( int j=0; j<24; j++ )
            {
                h = dbox( rc+s*rd, rb, 0.1 ); 
                s += h;
                if( s>tF ) break;
            }

            if( h < (surface*s*2.0) )
            {
                res = float3( s, cub.yz );
                break; 
            }
            
        }

        // step to next cell        
        mm = step( dis.xy, dis.yx ); 
        dis += mm*rda.xz;
        pos += mm*rds;
    }

    res.x += tmin;
    
    return res;
}

float usmoothstep( in float x )
{
    x = clamp(x,0.0,1.0);
    return x*x*(3.0-2.0*x);
}

float softshadow( in float3 ro, in float3 rd, in float mint, in float maxt, in float k )
{
    float res = 1.0;
    float t = mint;
    for( int i=0; i<50; i++ )
    {
        float h = map( ro + rd*t ).x;
        res = min( res, usmoothstep(k*h/t) );
        t += clamp( h, 0.05, 0.2 );
        if( res<0.001 || t>maxt ) break;
    }
    return clamp( res, 0.0, 1.0 );
}

float3 calcNormal( in float3 pos, in float t )
{
    float2 e = float2(1.0,-1.0)*surface*t;
    return normalize( e.xyy*map( pos + e.xyy ).x + 
                      e.yyx*map( pos + e.yyx ).x + 
                      e.yxy*map( pos + e.yxy ).x + 
                      e.xxx*map( pos + e.xxx ).x );
}

const float3 light1 = float3(  0.70, 0.52, -0.45 );
const float3 light2 = float3( -0.71, 0.000,  0.71 );
const float3 lpos = float3(0.0,0,0) + 6.0*float3(  0.70, 0.52, -0.45 );

float2 boundingVlume( float2 tminmax, in float3 ro, in float3 rd )
{
    float bp = 2.7;
    float tp = (bp-ro.y)/rd.y;
    if( tp>0.0 ) 
    {
        if( ro.y>bp ) tminmax.x = max( tminmax.x, tp );
        else          tminmax.y = min( tminmax.y, tp );
    }
    bp = 0.0;
    tp = (bp-ro.y)/rd.y;
    if( tp>0.0 ) 
    {
        if( ro.y>bp ) tminmax.y = min( tminmax.y, tp );
    }
    return tminmax;
}

float3 doLighting( in float3 col, in float ks,
                 in float3 pos, in float3 nor, in float3 rd )
{
    float3  ldif = lpos - pos;
    float llen = length( ldif );
    ldif /= llen;
    float con = dot( light1,ldif);
    float occ = lerp( clamp( pos.y/4.0, 0.0, 1.0 ), 1.0, 0.2*max(0.0,nor.y) );
    float2 sminmax = float2(0.01, 5.0);

    float sha = softshadow( pos, ldif, sminmax.x, sminmax.y, 32.0 );;
        
    float bb = smoothstep( 0.5, 0.8, con );
    float lkey = clamp( dot(nor,ldif), 0.0, 1.0 );
    float3  lkat = float3(1.0,1,1);
          lkat *= float3(bb*bb*0.6+0.4*bb,bb*0.5+0.5*bb*bb,bb).zyx;
          lkat /= 1.0+0.25*llen*llen;       
          lkat *= 30.0;
          //lkat *= sha;
          lkat *= float3(sha,0.6*sha+0.4*sha*sha,0.3*sha+0.7*sha*sha);
    
    float lbac = clamp( 0.5 + 0.5*dot( light2, nor ), 0.0, 1.0 );
          lbac *= smoothstep( 0.0, 0.8, con );
          lbac /= 1.0+0.2*llen*llen;        
          lbac *= 7.0;
    float lamb = 1.0 - 0.5*nor.y;
          lamb *= 1.0-smoothstep( 10.0, 25.0, length(pos.xz) );
          lamb *= 0.25 + 0.75*smoothstep( 0.0, 0.8, con );
          lamb *= 0.25;

    float3 lin  = 1.0*float3(1.60,0.70,0.30)*lkey*lkat*(0.5+0.5*occ);
         lin += 1.0*float3(0.20,0.05,0.02)*lamb*occ*occ;
         lin += 1.0*float3(0.70,0.20,0.08)*lbac*occ*occ;
         lin *= float3(1.3,1.1,1.0);
    
    col = col*lin;

    float3 hal = normalize(ldif-rd);
    float3 spe = lkey*lkat*(0.5+0.5*occ)*5.0*
               pow( clamp(dot(hal, nor),0.0,1.0), 6.0+6.0*ks ) * 
               (0.04+0.96*pow(clamp(1.0-dot(hal,ldif),0.0,1.0),5.0));

    col += (0.4+0.6*ks)*spe*float3(0.8,0.9,1.0);

    col = 1.4*col/(1.0+col);
    
    return col;
}

float3x3 setLookAt( in float3 ro, in float3 ta, float cr )
{
    float3  cw = normalize(ta-ro);
    float3  cp = float3(sin(cr), cos(cr),0.0);
    float3  cu = normalize( cross(cw,cp) );
    float3  cv = normalize( cross(cu,cw) );
    return float3x3( cu, cv, cw );
}

float3 render( in float3 ro, in float3 rd )
{
    float3 col = float3( 0.0,0,0 );

    float2 tminmax = float2(0.0, 40.0 );

    tminmax = boundingVlume( tminmax, ro, rd );

    // raytrace
    float3 res = trace( ro, rd, tminmax.x, tminmax.y );
    if( res.y > -0.5 )
    {
        float t = res.x;
        float3 pos = ro + t*rd;
        float3 nor = calcNormal( pos, t );

        // material 
        col = 0.5 + 0.5*cos( 6.2831*res.y + float3(0.0, 0.4, 0.8) );
        float3 ff = texcube( _MainTex, 0.21*float3(pos.x,4.0*res.z-pos.y,pos.z), nor ).xyz;
        ff = pow(ff,float3(1.3,1.3,1.3))*1.1;
        col *= ff.x;
col += ff;

        // lighting
//        col = doLighting( col, ff.x*ff.x*ff.x*2.0, pos, nor, rd );
        col *= 1.0 - smoothstep( 20.0, 40.0, t );
    }
    return col;
}


         fixed4 frag (v2f v) : SV_Target
            {
                float2 fragCoord = v.vertex;
                float2 screenUV = v.screenPos.xy / v.screenPos.w;

                float3 viewDirection = normalize(v.uv.xyz- _WorldSpaceCameraPos.xyz  );
                fixed4 fragColor = tex2D(_MainTex, v.uv);
                
                float3 rd = viewDirection;                                                        // ray direction for fragCoord.xy
                float3 ro = _WorldSpaceCameraPos.xyz+ _XYZPos;                                             // ray origin

    freqs[0] = _SoundArray[floor(.01*256)]/(.1+_Volume);  //tex2D( _MainTex, float2( 0.01, 0.25 ) ).x;
    freqs[1] = _SoundArray[floor(.07*256)]/(.1+_Volume);  //tex2D( _MainTex, float2( 0.07, 0.25 ) ).x;
    freqs[2] = _SoundArray[floor(.15*256)]/(.1+_Volume);  //tex2D( _MainTex, float2( 0.15, 0.25 ) ).x;
    freqs[3] = _SoundArray[floor(.30*256)]/(.1+_Volume);  //tex2D( _MainTex, float2( 0.30, 0.25 ) ).x;
    //-----------
    float time = 5.0 + 0.2*_Time.y;// + 20.0*iMouse.x/iResolution.x;
    
    float3 tot = float3(0.0,0,0);
    #if AA>1
    for( int j=0; j<AA; j++ )
    for( int i=0; i<AA; i++ )
    {
        float2 off = float2(float(i),float(j))/float(AA);
    #else
        float2 off = float2(0.0,0);
    #endif        
//        float2 xy = (-iResolution.xy+2.0*(fragCoord+off)) / iResolution.y;

        // camera   
//        float3 ro = float3( 8.5*cos(0.2+.33*time), 5.0+2.0*cos(0.1*time), 8.5*sin(0.1+0.37*time) );
        float3 ta = float3( -2.5+3.0*cos(1.2+.41*time), 0.0, 2.0+3.0*sin(2.0+0.38*time) );
        float roll = 0.2*sin(0.1*time);

        // camera tx
        float3x3 ca = setLookAt( ro, ta, roll );
  //      float3 rd = normalize( ca * float3(xy.xy,1.75) );
        //rd = normalize( mul(ca,rd) );

        float3 col = render( ro, rd );
        col = pow( col, float3(0.4545,0.4545,0.4545) );
        col = pow( col, float3(0.8,0.93,1.0) );
        //col = clamp(col,0.0,1.0);
        tot += col;
        
    #if AA>1
    }
    tot /= float(AA*AA);
    #endif    
    
    // vigneting
 //   float2 q = fragCoord.xy/iResolution.xy;
 //   tot *= 0.2 + 0.8*pow( 16.0*q.x*q.y*(1.0-q.x)*(1.0-q.y), 0.1 );

    fragColor = float4( tot, 1.0 );


                return fragColor;
            }

            ENDCG
        }
    }
}



