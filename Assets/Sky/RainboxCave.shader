
Shader "Skybox/RainboxCave"
{
    Properties
    {
        _MainTex ("Texture", 2D) = "white" {}
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


float2 path(float z){
    return float2(
        .01*sin(z*40.),
        .03*cos(z*21.)+.5*sin(cos(z))
    );
}



float g(float3 p){

    float2 q=path(p.z);
    float3 n=frac(p)-.5;
    float m,s=1.,a=.01,b=.001,t=max(abs(p.x+q.x),abs(p.y+q.y));

    for(int i=1;i<9;++i){
        n/=m=dot(n,n)*.7;
        n.xy=frac(n.xy)-.5;
        s*=m;
        n.xyz=n.yzx;
    };

    s=min(9.,(length(n)-1.)*s);

    m=clamp(.3+.5*s/.05,a,1.);

    m=lerp(-t,-s,m+t)+m*.05*sin(1.);
    p.xy+=path(p.z);
    return min(
        max(
            m,
            a-max(m=abs(p.x),abs(p.y))
        ),
        max(
            m-b,
            abs(p.y+=a)-b
        )
    );
}


float3 f(float i){
    i+=_Time.y/20.;
    return float3(-path(i),i);
}

         fixed4 frag (v2f v) : SV_Target
            {
                float2 b = v.vertex;

                float3 viewDirection = normalize(v.uv.xyz- _WorldSpaceCameraPos.xyz  );
                fixed4 a = tex2D(_MainTex, v.uv);
                
                float3 rd = viewDirection;                                                        // ray direction for fragCoord.xy
                float3 ro = _WorldSpaceCameraPos.xyz*.0001;                                             // ray origin


    float d,t=0.,n=.0001;

    float2 q=float2(1,1);  //2.*(b/iResolution.xy)-1.;

    float3 c,p,o=f(t),
    w=normalize(f(.01)-o),
    r=cross(w,float3(n,1.,t));

//    r=float3x3(
//        r,
//        normalize(cross(r,w)),
//        w
//    )*normalize(
//        float3(q.xy,1.5)
//    );

    r = rd;

    float4 K=float4(1.,.7,.3,3.);

    for(int i=0;i<128;++i){
        d=g(p=o+r*t);
        if(d<n){break;};
        t+=d*.5;
    };
    q=float2(.0,n);

    d=max(n,dot(-r,normalize(
        float3(
            g(p+(c=q.yxx))-g(p-c),
            g(p+(c=q.xyx))-g(p-c),
            g(p+(c=q.xxy))-g(p-c)
        )
    ))*.5+.4);



    c=float3(
        22.*lerp(t=p.y,t,n),
        n,
        t
    );

    a=float4(lerp(
        lerp(
            p=K.xyz*cos(t)*cos(p.y),
            clamp(
                abs(frac(c.xxx+p)*5.-3.)-p,
                t,
                1.
            ),
            .5
        )*d,
        c,
        n
    ),t);

                return a;
            }

            ENDCG
        }
    }
}

